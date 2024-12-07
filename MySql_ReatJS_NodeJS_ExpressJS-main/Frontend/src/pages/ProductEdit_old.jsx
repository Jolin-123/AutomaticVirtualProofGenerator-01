//rfce
//2/29/2024
import React from 'react'
import '../App.css'
import { useNavigate} from 'react-router-dom';
import axios from 'axios';
import  { useEffect, useState} from 'react'
import { Link } from 'react-router-dom'


function ProductEdit() {

        const [ord, setOrd] = useState('');
        const [customer, setCustomer] = useState('');
        const [label, setLabel] = useState('');
        const [bid, setBid] = useState('');
        const [gid, setGid] = useState('');
        const [bcolor, setBcolor] = useState('');
        const [team, setTeam] = useState('');
        const [bdesc, setBdesc] = useState('');

        const [xs, setXs] = useState(0);
        const [s, setS] = useState(0); // Initialize with 0 as an example
        const [m, setM] = useState(0);
        const [l, setL] = useState(0);
        const [xl, setXl] = useState(0);
        const [xxl, setXxl] = useState(0);
        const [xxxl, setXxxl] = useState(0);
        const [ttl, setTtl] = useState(0);
        const [start, setStart] = useState('');
        const [end, setEnd] = useState('');

        const [oid, setOid] = useState(0); // State variable to store the filename
        const { idx: RouterId } = useParams(); // Extract idx from URL params
    
        useEffect(() => {
            fetchData();
            setOid(RouterId);
            console.log("This list id is: " , RouterId);
        }, [RouterId]); // Fetch data whenever oid changes
    
        // might need it, might not need it
        // const fetchData = () => {
        //     axios.get(`http://localhost:3030/oneOrder/${RouterId}/`)
        //         .then(res => {
        //             if (res.data) {
        //                 setOrd(res.ord);
        //                 setCustomer(res.data.customer);
        //                 setLabel(res.data.label);
        //                 setBid(res.data.bid);
        //                 setGid(res.data.gid);
        //                 setBcolor(res.data.bcolor);
        //                 setTeam(res.data.team);
        //                 setDesc(res.data.bdesc);

        //                 setXs(res.data.xs);
        //                 setS(res.data.s);
        //                 setM(res.data.m);
        //                 setL(res.data.l);
        //                 setXl(res.data.xl);
        //                 setXxl(res.data.xxl);
        //                 setXxxl(res.data.xxxl);
        //                 setTtl(res.data.ttl);
        //                 setStart(res.data.start);
        //                 setEnd(res.data.end);
        //             }
        //         })
        //         .catch(err => console.log(err));
        // };
    
    
        const navigate = useNavigate();
    
        function handleSubmit(event) {
            event.preventDefault();
        
            // Check if any of the required fields are empty
            if (!ord || !customer || !label || !bid || !gid || !bcolor || !team || !bdesc || !xs || !s || !m || !l || !xl || !xxl || !xxxl || !ttl || !start || !end) {
                console.log("Please fill in all required fields.");
                window.alert("Please fill in all required fields.");
                return;
            }
        
            // All required fields are filled, proceed with the POST request
            event.preventDefault();
            axios.put(`http://localhost:3030/update/${oid}/`, { ord, customer, label, bid, gid, bcolor, team, bdesc, xs, s, m, l, xl, xxl, xxxl, ttl, start, end })
                .then(res => {
                    console.log(res);
                    navigate('/ProductList'); //After click Update button, jump to ProductList page
            })
            .catch(err => console.log("Error comes out when PUT/edit the info of product @@@ "));
    }
        
    
      return (
        <div> 
            <Link to="/productList" className='btn update' style={{  backgroundColor: "lightblue" , color: 'gray' }}> {'<-'} Back </Link>
            <div className='d-flex vh-100 bg-primary justify-content-center align-items-center'>
                <div className='add_order_table '>
    
                <form onSubmit={handleSubmit}  className='form-contro_big'>
                    <h2 className='add_order_title' style={{color: "green"}}> Product Edit  </h2>
    
                    <div className="add_order_element">
                        <label htmlFor="ord"> Order ID </label>
                        <input
                            type="text"
                            className="form-control"
                            placeholder='Enter Order ID'
                            value={ord}
                            onChange={e => setOrd(e.target.value)}
                        />
                    </div>
    
    
                    <div className="add_order_element">
                        <label htmlFor="customer"> Customer</label>
                        <input
                            type="text"
                            className="form-control"
                            // id="customer"
                            // name="customer"
                            placeholder='Enter Customer Name'
                            value={customer}
                            onChange={e => setCustomer(e.target.value)}
                        />
                    </div>
    
                    
                    <div className="add_order_element">
                        <label htmlFor="label"> Label </label>
                        <input
                            type="text"
                            className="form-control"
                            // id="label"
                            // name="label"
                            placeholder='Enter Label Name'
                            value={label}
                            onChange={e => setLabel(e.target.value)}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label htmlFor="bid"> Body ID </label>
                        <input
                            type="text"
                            className="form-control"
                            // id="bid"
                            // name="bid"
                            placeholder='Enter Body ID'
                            value={bid}
                            onChange={e => setBid(e.target.value)}
                        />
                    </div>
    
    
                    <div className="add_order_element">
                        <label htmlFor="gid"> Graphic ID </label>
                        <input
                            type="text"
                            className="form-control"
                            // id="gid"
                            // name="gid"
                            placeholder='Enter Graphic ID'
                            value={gid}
                            onChange={e => setGid(e.target.value)}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label htmlFor="bcolor"> Body Color </label>
                        <input
                            type="text"
                            className="form-control"
                            // id="bcolor"
                            // name="bcolor"
                            placeholder='Enter Body Color'
                            value={bcolor}
                            onChange={e => setBcolor(e.target.value)}
                        />
                    </div>
                    
                
                    <div className="add_order_element">
                        <label htmlFor="team"> Team Name </label>
                        <input
                            type="text"
                            className="form-control"
                            // id="team"
                            // name="team"
                            placeholder='Enter Team Name'
                            value={team}
                            onChange={e => setTeam(e.target.value)}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label htmlFor="bdesc"> Body Desc </label>
                        <input
                            type="text"
                            className="form-control"
                            // id="bdesc"
                            // name="bdesc"
                            placeholder='Enter Body Desc'
                            value={bdesc}
                            onChange={e => setBdesc(e.target.value)}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label htmlFor="xs"> XS </label>
                        <input
                            type="number"
                            className="form-control"
                            // id="xs"
                            // name="xs"
                            placeholder='Enter xs'
                            value={xs}
                            onChange={e => setXs(e.target.value)}
                        />
                    </div>
    
    
                    <div className="add_order_element">
                        <label htmlFor="s"> S </label>
                        <input
                            type="number"
                            className="form-control"
                            // id="s"
                            // name="s"
                            placeholder='Enter s'
                            value={s}
                            //onChange={e => setS(e.target.value)}
                            onChange={e => setS(parseInt(e.target.value))}
                        />
                    </div>
    
                    
                    <div className="add_order_element">
                        <label htmlFor="m"> M </label>
                        <input
                            type="number"
                            className="form-control"
                            // id="m"
                            // name="m"
                            placeholder='Enter m'
                            value={m}
                            //onChange={e => setM(e.target.value)}
                            onChange={e => setM(parseInt(e.target.value))}
                        />
                    </div>
    
    
                    <div className="add_order_element">
                        <label for="l"> L </label>
                        <input
                            type="number"
                            class="form-control"
                            id="l"
                            name="l"
                            placeholder="Enter L"
                            value={l}
                            //onChange={e => setL(e.target.value)}
                            onChange={e => setL(parseInt(e.target.value))}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label for="xl">XL </label>
                        <input
                            type="number"
                            class="form-control"
                            id="xl"
                            name="xl"
                            placeholder="Enter XL"
                            value={xl}
                            //onChange={e => setXl(e.target.value)}
                            onChange={e => setXl(parseInt(e.target.value))}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label for="xxl">XXL </label>
                        <input
                            type="number"
                            class="form-control"
                            id="xxl"
                            name="xxl"
                            placeholder="Enter XXL"
                            value={xxl}
                            //onChange={e => setXxl(e.target.value)}
                            onChange={e => setXxl(parseInt(e.target.value))}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label for="xxxl">XXXL </label>
                        <input
                            type="number"
                            class="form-control"
                            id="xxxl"
                            name="xxxl"
                            placeholder="Enter XXXL"
                            value={xxxl}
                            //onChange={e => setXxxl(e.target.value)}
                            onChange={e => setXxxl(parseInt(e.target.value))}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label for="ttl">Total </label>
                        <input
                            type="number"
                            class="form-control"
                            id="ttl"
                            name="ttl"
                            placeholder="Enter Total"
                            value={ttl}
                            //onChange={e => setTtl(e.target.value)}
                            onChange={e => setTtl(parseInt(e.target.value))}
                        />
                    </div>
    
    
                    <div className="add_order_element">
                        <label for="start">Start Date </label>
                        <input
                            type="date"
                            class="form-control"
                            id="start"
                            name="start"
                            placeholder="Enter Start Date"
                            value={start}
                            onChange={e => setStart(e.target.value)}
                        />
                    </div>
    
                    <div className="add_order_element">
                        <label for="end">End Date </label>
                        <input
                            type="date"
                            class="form-control"
                            id="end"
                            name="end"
                            placeholder="Enter End Date"
                            value={end}
                            onChange={e => setEnd(e.target.value)}
                        />
                    </div>
    
                       
                        <button type="submit" className=" btn-submit"> Update </button>
                      
                    </form>
            </div>
            </div>
        </div>
      )
    
    }

export default ProductEdit;