// Frontend code
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import Sidebar from '../components/Sidebar'


function Comment() {
    const [tshirts, setTshirts] = useState([]);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = () => {
        axios.get('http://localhost:3030/alltshirt/')
            .then(res => {
                setTshirts(res.data);
            })
            .catch(err => console.log(err));
    };

    const handleDelete = async (idx) => {
      try {
          await axios.delete(`http://localhost:3030/deleteTshirtInfo/${idx}`); //call the api
          window.location.reload();
      }catch(err){
          console.log("err from delete tshirt list frontend @@ ");
      }
    }

    
    return (

        <Sidebar>
        <div className='container_main  logotable_display'>
            <div className='table_container'>
                <h2 className='order_list_title'> Tshirt List ~ </h2>
                <table className='list_table table_row'>
                    <thead>
                        <tr>
                            <th>Tshirt ID</th>
                            <th>Style</th>
                            <th>Color</th>
                            <th>Description</th>
                            <th>Price</th>
                            <th>Image</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tshirts.map((d, i) => (
                            <tr key={i}>
                                <td>{d.idx}</td>
                                <td>{d.style}</td>
                                <td>{d.color}</td>
                                <td>{d.description}</td>
                                <td>{d.price}</td>
                                {/* Display image */}
                                <td>
                                    {d.image && <img src={`http://localhost:3030/images/${d.image}`} alt="Tshirt" style={{ width: "100px", height: "100px" }} />}
                                </td>
                                <td className='button-container'>
                                    {/* <button className='btn read'>Read</button> */}
                                    <Link to={`/updatetshirt/${d.idx}`} className='btn update'>Update</Link>
                                    <button className='btn delete' onClick={e => handleDelete(d.idx)}>Delete</button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
        </Sidebar>
    );
}

export default Comment;
