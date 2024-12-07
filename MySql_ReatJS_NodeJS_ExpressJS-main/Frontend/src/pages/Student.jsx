import React, { useEffect, useState} from 'react'
import axios from 'axios';
import './Student.css' ;
import { Link } from 'react-router-dom'


function Student() {

    const [data, setData] = useState([]);
  
    //************** 1/28/2024 ************** ************** ************** **************
    // Formating data to "YYYY-MM-DD" format
      function formatDate(dateString) {
        const date = new Date(dateString);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    useEffect(() => {
        axios.get('http://localhost:8081/users')
            .then(res => {
                const formattedData = res.data.map(item => ({
                    ...item,
                    date: formatDate(item.date)
                }));
                setData(formattedData);
            })
            .catch(err => console.log(err));
    }, []);

    
      const handleDelete = async (id) => {
        try {
            await axios.delete(`http://localhost:8081/users/${id}`); //call the api
            window.location.reload();
        }catch(err){
            console.log("err from delete @@ ");
        }
      }


    return (
        <div className=''> 
        
             <div className='w-50 bg-white rounded p-3 '>
                <Link to="/create" className='btn btn-success'>
                    Add + 
                </Link>

                <Link to="/create" className='btn btn-success'>
                    Dash Board
                </Link>

                <table className='table custom-table'>  
                    <thead>
                    <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Phone</th>
                    <th>Email</th>
                    <th>Date</th>
                    <th>Action</th>
                    </tr>
                    </thead>
                    
                    <tbody>
                    {
                        data.map((d, i) => (
                            <tr key={i}>
                                <td>{d.id}</td>
                                <td>{d.name}</td>
                                <td>{d.phone}</td>
                                <td>{d.email}</td>
                                <td>{d.date}</td>

                            <td>
                                <button className='table_btn read'> Read </button>
                                <Link to={`update/${d.id}`}  className='btn update'> Update </Link>
                                <button className='table_btn delete' onClick={e=> handleDelete(d.id)}> Delete </button>
                            </td>
                        </tr>
                      ))
                      }

                    </tbody>
                 </table>
             
             </div>
        </div>
    );
}

export default Student; 